import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';

const useStyles = makeStyles((theme) => ({
  root: {
    width: '100%',
  },
  bullet: {
    display: 'inline-block',
    margin: '0 2px',
    transform: 'scale(0.8)',
  },
  title: {
    fontSize: 14,
  },
  pos: {
    marginBottom: 12,
  },
  sentence: {
    ...theme.typography.body1,
  },
  keySentence: {
    ...theme.typography.body1,
    backgroundColor: "rgba(255, 229, 100, 0.2)"
  },
}));

export default function NewsCard(props) {
  const classes = useStyles();
  const bull = <span className={classes.bullet}>•</span>;

  return (
    <Card className={classes.root} variant="outlined">
      <CardContent>
        <Typography className={classes.title} color="textSecondary" gutterBottom>
          {props.companyName}
        </Typography>
        <Typography variant="h5" component="h2">
          {props.title}
        </Typography>
        <Typography className={classes.pos} color="textSecondary">
          {props.date}
        </Typography>
        <Typography variant="body2" component="p">
          {props.paragraph != undefined &&
            <div>
              {props.paragraph.map((item, index) =>
                item.isKeySentence ?
                  <p key={index} className={classes.keySentence}>
                    {item.text}
                  </p>
                  :
                  <p key={index} className={classes.sentence}>
                    {item.text}
                  </p>
              )}
            </div>
          }
        </Typography>
      </CardContent>
      {/* <CardActions>
        <Button size="small">Learn More</Button>
      </CardActions> */}
    </Card>
  );
}