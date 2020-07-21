import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Dialog from '@material-ui/core/Dialog';
import DialogContent from '@material-ui/core/DialogContent';
import Typography from '@material-ui/core/Typography';


const useStyles = makeStyles((theme) => ({
  successText: {
    textAlign: "center",
    margin: theme.spacing(0, 0, 2),
  }
}));
  
export default function MessageDialog(props) {

  const classes = useStyles();

  return (
    <div>
      <Dialog open={props.isOpen} onClose={props.handleClose} aria-labelledby="form-dialog-title">
        <DialogContent>
          <Typography className={classes.successText}>
            {props.message}
          </Typography>
        </DialogContent>
      </Dialog>
    </div>
  );
}